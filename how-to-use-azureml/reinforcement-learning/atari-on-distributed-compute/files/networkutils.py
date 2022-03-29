import sys
import csv
from azure.mgmt.network import NetworkManagementClient


def check_port_in_port_range(expected_port: str,
                             dest_port_range: str):
    """
    Check if a port is within a port range
    Port range maybe like *, 8080 or 8888-8889
    """

    if dest_port_range == '*':
        return True

    dest_ports = dest_port_range.split('-')

    if len(dest_ports) == 1 and \
       int(dest_ports[0]) == int(expected_port):
        return True

    if len(dest_ports) == 2 and \
       int(dest_ports[0]) <= int(expected_port) and \
       int(dest_ports[1]) >= int(expected_port):
        return True

    return False


def check_port_in_destination_port_ranges(expected_port: str,
                                          dest_port_ranges: list):
    """
    Check if a port is within a given list of port ranges
    i.e. check if port 8080 is in port ranges of 22,80,8080-8090,443
    """

    for dest_port_range in dest_port_ranges:
        if check_port_in_port_range(expected_port, dest_port_range) is True:
            return True

    return False


def check_ports_in_destination_port_ranges(expected_ports: list,
                                           dest_port_ranges: list):
    """
    Check if all ports in a given port list are within a given list
    of port ranges
    i.e. check if port 8080,8081 are in port ranges of 22,80,8080-8090,443
    """

    for expected_port in expected_ports:
        if check_port_in_destination_port_ranges(
           expected_port, dest_port_ranges) is False:
            return False

    return True


def check_source_address_prefix(source_address_prefix: str):
    """Check if source address prefix is BatchNodeManagement or default"""

    required_prefix = 'BatchNodeManagement'
    default_prefix = 'default'

    if source_address_prefix.lower() == required_prefix.lower() or \
       source_address_prefix.lower() == default_prefix.lower():
        return True

    return False


def check_protocol(protocol: str):
    """Check if protocol is supported - Tcp/Any"""

    required_protocol = 'Tcp'
    any_protocol = 'Any'

    if required_protocol.lower() == protocol.lower() or \
       any_protocol.lower() == protocol.lower():
        return True

    return False


def check_direction(direction: str):
    """Check if port direction is inbound"""

    required_direction = 'Inbound'

    if required_direction.lower() == direction.lower():
        return True

    return False


def check_provisioning_state(provisioning_state: str):
    """Check if the provisioning state is succeeded"""

    required_provisioning_state = 'Succeeded'

    if required_provisioning_state.lower() == provisioning_state.lower():
        return True

    return False


def check_rule_for_Azure_ML(rule):
    """Check if the ports required for Azure Machine Learning are open"""

    required_ports = ['29876', '29877']

    if check_source_address_prefix(rule.source_address_prefix) is False:
        return False

    if check_protocol(rule.protocol) is False:
        return False

    if check_direction(rule.direction) is False:
        return False

    if check_provisioning_state(rule.provisioning_state) is False:
        return False

    if rule.destination_port_range is not None:
        if check_ports_in_destination_port_ranges(
           required_ports,
           [rule.destination_port_range]) is False:
            return False
    else:
        if check_ports_in_destination_port_ranges(
           required_ports,
           rule.destination_port_ranges) is False:
            return False

    return True


def check_vnet_security_rules(auth_object,
                              vnet_subscription_id,
                              vnet_resource_group,
                              vnet_name,
                              save_to_file=False):
    """
    Check all the rules of virtual network if required ports for Azure Machine
    Learning are open
    """

    network_client = NetworkManagementClient(
        auth_object,
        vnet_subscription_id)

    # get the vnet
    vnet = network_client.virtual_networks.get(
        resource_group_name=vnet_resource_group,
        virtual_network_name=vnet_name)

    vnet_location = vnet.location
    vnet_info = []

    if vnet.subnets is None or len(vnet.subnets) == 0:
        print('WARNING: No subnet found for VNet:', vnet_name)

    # for each subnet of the vnet
    for subnet in vnet.subnets:
        if subnet.network_security_group is None:
            print('WARNING: No network security group found for subnet.',
                  'Subnet',
                  subnet.id.split("/")[-1])
        else:
            # get all the rules
            network_security_group_name = \
                subnet.network_security_group.id.split("/")[-1]
            network_security_group_resource_group_name = \
                subnet.network_security_group.id.split("/")[4]
            network_security_group_subscription_id = \
                subnet.network_security_group.id.split("/")[2]

            security_rules = list(network_client.security_rules.list(
                network_security_group_resource_group_name,
                network_security_group_name))

            rule_matched = None
            for rule in security_rules:
                rule_info = []
                # add vnet details
                rule_info.append(vnet_name)
                rule_info.append(vnet_subscription_id)
                rule_info.append(vnet_resource_group)
                rule_info.append(vnet_location)
                # add subnet details
                rule_info.append(subnet.id.split("/")[-1])
                rule_info.append(network_security_group_name)
                rule_info.append(network_security_group_subscription_id)
                rule_info.append(network_security_group_resource_group_name)
                # add rule details
                rule_info.append(rule.priority)
                rule_info.append(rule.name)
                rule_info.append(rule.source_address_prefix)
                if rule.destination_port_range is not None:
                    rule_info.append(rule.destination_port_range)
                else:
                    rule_info.append(rule.destination_port_ranges)
                rule_info.append(rule.direction)
                rule_info.append(rule.provisioning_state)
                vnet_info.append(rule_info)

                if check_rule_for_Azure_ML(rule) is True:
                    rule_matched = rule

            if rule_matched is not None:
                print("INFORMATION: Rule matched with required ports. Subnet:",
                      subnet.id.split("/")[-1], "Rule:", rule.name)
            else:
                print("WARNING: No rule matched with required ports. Subnet:",
                      subnet.id.split("/")[-1])

    if save_to_file is True:
        file_name = vnet_name + ".csv"
        with open(file_name, mode='w') as vnet_rule_file:
            vnet_rule_file_writer = csv.writer(
                vnet_rule_file,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            header = ['VNet_Name', 'VNet_Subscription_ID',
                      'VNet_Resource_Group', 'VNet_Location',
                      'Subnet_Name', 'NSG_Name',
                      'NSG_Subscription_ID', 'NSG_Resource_Group',
                      'Rule_Priority', 'Rule_Name', 'Rule_Source',
                      'Rule_Destination_Ports', 'Rule_Direction',
                      'Rule_Provisioning_State']
            vnet_rule_file_writer.writerow(header)
            vnet_rule_file_writer.writerows(vnet_info)

        print("INFORMATION: Network security group rules for your virtual \
network are saved in file", file_name)
